import gldac

import numpy as np
import pandas as pd
import typing
import os
from sklearn.feature_extraction.text import CountVectorizer

from d3m.primitive_interfaces.unsupervised_learning import UnsupervisedLearnerPrimitiveBase
from d3m.primitive_interfaces import base
from d3m import container, utils
from d3m.metadata import hyperparams, base as metadata_base
from d3m.metadata import params

from gensim.models import Word2Vec
from fastlvm.utils import get_documents, mk_text_features, tokenize, split_inputs

Inputs = container.DataFrame
Outputs = container.DataFrame
Predicts = container.ndarray  # type: np.ndarray


class Params(params.Params):
    topic_matrix: bytes  # Byte stream represening topics


class HyperParams(hyperparams.Hyperparams):
    k = hyperparams.UniformInt(lower=1, upper=10000, default=10,
                               semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
                               description='The number of clusters to form as well as the number of centroids to '
                                           'generate.')
    iters = hyperparams.UniformInt(lower=1, upper=10000, default=100,
                                   semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
                                   description='The number of iterations of inference.')
    num_top = hyperparams.UniformInt(lower=1, upper=10000, default=1,
                                     semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
                                     description='The number of top words requested')
    w2v_size = hyperparams.Hyperparameter[int](
        default=30,
        description="Dimensionality of the feature vectors.",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    w2v_window = hyperparams.Hyperparameter[int](
        default=5,
        description="The maximum distance between the current and predicted word within a sentence.",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    w2v_min_count = hyperparams.Hyperparameter[int](
        default=1,
        description="The mininmum count",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    w2v_iters = hyperparams.Hyperparameter[int](
        default=30,
        description='Number of iterations (epochs) over the corpus.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    frac = hyperparams.Uniform(lower=0, upper=1, default=0.01, upper_inclusive=False,
                               semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
                               description='The fraction of training data set aside as the validation. 0 = use all '
                                           'training as validation')
    seed = hyperparams.UniformInt(lower=-1000000, upper=1000000, default=1,
                                  semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
                                  description='A random seed to use')


class GLDA(UnsupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, HyperParams]):
    """
    This class provides functionality for unsupervised inference on Gaussian latent Dirichlet allocation,
    which replace LDA's parameterization of 'topics' as categorical distributions over opaque word types with
    multivariate Gaussian distributions on the embedding space. This encourages the model to group words that are a
    priori known to be semantically related into topics, as continuous space word embeddings learned from large,
    unstructured corpora have been shown to be effective at capturing semantic regularities in language. Using
    vectors learned from a domain-general corpus (e.g. English Wikipedia), qualitatively, Gaussian LDA infers
    different (but still very sensible) topics relative to standard LDA. Quantitatively, the technique outperforms
    existing models at dealing with OOV words in held-out documents. No standard packages exists. Our underlying C++
    implementation can be distributed to run on multiple machines. To enable the distribution through python
    interface is work in progress. In this class, we implement inference on Gaussian latent Dirichlet Allocation
    using Canopy algorithm. In case of full covariance matrices, it exploits the Cholesky decompositions of
    covariance matrices of the posterior predictive distributions and performs efficient rank-one updates. The API is
    similar to sklearn.decomposition.LatentDirichletAllocation.
    """

    metadata = metadata_base.PrimitiveMetadata({
        "id": "a3d490a4-ef39-4de1-be02-4c43726b3b24",
        "version": "3.1.1",
        "name": "Gaussian Latent Dirichlet Allocation Topic Modelling",
        "description": "This class provides functionality for unsupervised inference on Gaussian latent Dirichlet "
                       "allocation, which replace LDA's parameterization of 'topics' as categorical distributions "
                       "over opaque word types with multivariate Gaussian distributions on the embedding space. This "
                       "encourages the model to group words that are a priori known to be semantically related into "
                       "topics, as continuous space word embeddings learned from large, unstructured corpora have "
                       "been shown to be effective at capturing semantic regularities in language. Using vectors "
                       "learned from a domain-general corpus (e.g. English Wikipedia), qualitatively, Gaussian LDA "
                       "infers different (but still very sensible) topics relative to standard LDA. Quantitatively, "
                       "the technique outperforms existing models at dealing with OOV words in held-out documents. No "
                       "standard packages exists. Our underlying C++ implementation can be distributed to run on "
                       "multiple machines. To enable the distribution through python interface is work in progress. "
                       "In this class, we implement inference on Gaussian latent Dirichlet Allocation using Canopy "
                       "algorithm. In case of full covariance matrices, it exploits the Cholesky decompositions of "
                       "covariance matrices of the posterior predictive distributions and performs efficient rank-one "
                       "updates. The API is similar to sklearn.decomposition.LatentDirichletAllocation.",
        "python_path": "d3m.primitives.natural_language_processing.glda.Fastlvm",
        "primitive_family": metadata_base.PrimitiveFamily.NATURAL_LANGUAGE_PROCESSING,
        "algorithm_types": ["LATENT_DIRICHLET_ALLOCATION"],
        "keywords": ["large scale Gaussian LDA", "topic modeling", "clustering"],
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
        # super(GLDA, self).__init__()
        super().__init__(hyperparams=hyperparams)
        self._this = None
        self._k = hyperparams['k']
        self._iters = hyperparams['iters']
        self._num_top = hyperparams['num_top']
        self._frac = hyperparams['frac']  # the fraction of training data set aside as the validation
        self._seed = hyperparams['seed']

        self._training_inputs = None  # type: Inputs
        self._validation_inputs = None  # type: Inputs
        self._fitted = False
        self._ext = None
        self._vectorizer = None  # for tokenization
        self._analyze = None  # to tokenize raw documents

        self._w2v_size = hyperparams['w2v_size']
        self._w2v_window = hyperparams['w2v_window']
        self._w2v_min_count = hyperparams['w2v_min_count']
        self._w2v_iters = hyperparams['w2v_iters']
        self.hyperparams = hyperparams

    def __del__(self):
        if self._this is not None:
            gldac.delete(self._this, self._ext)

    def set_training_data(self, *, inputs: Inputs) -> None:
        """
        Sets training data for GLDA.

        Parameters
        ----------
        inputs : Inputs
            A list of 1d numpy array of dtype uint32. Each numpy array contains a document with each token mapped to its word id.
        """

        self._training_inputs = inputs

        self._fitted = False

    def fit(self, *, timeout: float = None, iterations: int = None) -> base.CallResult[None]:
        """
        Inference on the Gaussian latent Dirichley allocation model
        """
        if self._fitted:
            return

        if self._training_inputs is None:
            raise ValueError("Missing training data.")

        # FIXME Moved the fit implementation into the produce().
        # Calling fit here and glabc.predict() in produce causes SIGSEGV.

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
        return self._fitted

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
        """
        Finds the token topic assignment (and consequently topic-per-document distribution) for the given set of docs
        using the learned model.

        Parameters
        ----------
        inputs : Inputs
            A list of 1d numpy array of dtype uint32. Each numpy array contains a document with each token mapped to
            its word id.

        Returns
        -------
        Outputs
            A list of 1d numpy array which represents index of the topic each token belongs to.

        """

        # ============================================================
        # Start of fit()
        # ============================================================
        # Create documents from the data-frame
        raw_documents = get_documents(self._training_inputs)

        if raw_documents is None:  # training data contains no text fields
            if self._this is not None:
                gldac.delete(self._this, self._ext)
            self._this = None
            return base.CallResult(inputs)

        # Extract the vocabulary from the inputs data-frame
        self._vectorizer = CountVectorizer()
        self._vectorizer.fit(raw_documents)
        vocab_size = len(self._vectorizer.vocabulary_)
        vocab = ['w' + str(i) for i in range(vocab_size)]

        # Build analyzer that handles tokenization
        self._analyze = self._vectorizer.build_analyzer()

        # Represent the documents in w2v
        size = self._w2v_size
        word_list = []
        for doc in raw_documents:
            # Consider using self._analyze
            words = doc.split()
            word_list.append(words)
        w2v = Word2Vec(word_list, size=size, window=self._w2v_window, min_count=self._w2v_min_count,
                       iter=self._w2v_iters)

        # Create the vocabulary using word2vec
        wv = np.zeros((vocab_size, size))
        for w, i in self._vectorizer.vocabulary_.items():
            if w in w2v.wv:
                wv[i] = w2v.wv[w]
            else:
                wv[i] = 2 * np.random.randn(size)

        # Tokenize documents
        tokenized = tokenize(raw_documents, self._vectorizer.vocabulary_, self._analyze)

        # Uniformly split the data to training and validation
        training, validation = split_inputs(tokenized, self._frac)

        # Release the old object to prevent memory leaking
        self.__del__()
        self._this = gldac.new(self._k, self._iters, vocab, wv)
        gldac.fit(self._this, training.tolist(), validation.tolist())
        # ============================================================
        # End of fit()
        # ============================================================

        # Get per-word topic assignment
        raw_documents, non_text_features = get_documents(inputs, non_text=True)
        tokenized = tokenize(raw_documents, self._vectorizer.vocabulary_, self._analyze)
        predicted = gldac.predict(self._this, tokenized.tolist())  # per word topic assignment
        text_features = mk_text_features(predicted, self._k)

        # concatenate the features row-wise
        features = pd.concat([non_text_features, text_features], axis=1)

        # append columns in the metadata
        features.metadata = features.metadata.append_columns(text_features.metadata)

        return base.CallResult(features)

    def evaluate(self, *, inputs: Inputs) -> float:
        """
        Finds the per-token log likelihood (-ve log perplexity) of learned model on a set of test docs.
        
        Parameters
        ----------
        inputs : Inputs
            A list of 1d numpy array of dtype uint32. Each numpy array contains a document with each token mapped to
            its word id. This represents test docs to test the learned model.

        Returns
        -------
        score : float
            Final per-token log likelihood (-ve log perplexity).
        """
        return gldac.evaluate(self._this, inputs)

    def produce_top_words(self) -> Outputs:
        """
        Get the top words of each topic for this model.

        Returns
        ----------
        topic_matrix : list
            A list of size k containing list of size num_top words.
        """

        return gldac.top_words(self._this, self._num_top)

    def produce_topic_matrix(self) -> np.ndarray:
        """
        Get current word|topic distribution matrix for this model.

        Returns
        ----------
        topic_matrix : numpy.ndarray
            A numpy array of shape (vocab_size,k) with each column containing the word|topic distribution.
        """

        if self._ext is None:
            self._ext = gldac.topic_matrix(self._this)
        return self._ext

    def multi_produce(self, *, produce_methods: typing.Sequence[str], inputs: Inputs, timeout: float = None,
                      iterations: int = None) -> base.MultiCallResult:
        return self._multi_produce(produce_methods=produce_methods, timeout=timeout, iterations=iterations,
                                   inputs=inputs)

    def get_params(self) -> Params:
        """
        Get parameters of GLDA.

        Parameters are basically the topic matrix in byte stream.

        Returns
        ----------
        params : Params
            A named tuple of parameters.
        """

        return Params(topic_matrix=gldac.serialize(self._this))

    def set_params(self, *, params: Params) -> None:
        """
        Set parameters of LDA.

        Parameters are basically the topic matrix in byte stream.

        Parameters
        ----------
        params : Params
            A named tuple of parameters.
        """
        self._this = gldac.deserialize(params['topic_matrix'])

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
