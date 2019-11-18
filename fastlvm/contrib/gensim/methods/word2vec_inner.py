# encoding: utf-8
# module gensim.models.word2vec_inner
# from /data/apps/anaconda3/envs/d3m-v2019.11.10/lib/python3.6/site-packages/gensim/models/word2vec_inner.cpython-36m-x86_64-linux-gnu.so
# by generator 1.147
""" Optimized cython functions for training :class:`~gensim.models.word2vec.Word2Vec` model. """

# imports
import builtins as __builtins__ # <module 'builtins' (built-in)>
import numpy as np # /data/apps/anaconda3/envs/d3m/lib/python3.6/site-packages/numpy/__init__.py
import scipy.linalg.blas as fblas # /data/apps/anaconda3/envs/d3m/lib/python3.6/site-packages/scipy/linalg/blas.py
import numpy as __numpy


# Variables with simple values

FAST_VERSION = 1

MAX_WORDS_IN_BATCH = 10000

# functions

def init(): # real signature unknown; restored from __doc__
    """
    init()
    Precompute function `sigmoid(x) = 1 / (1 + exp(-x))`, for x values discretized into table EXP_TABLE.
         Also calculate log(sigmoid(x)) into LOG_TABLE.
    
        Returns
        -------
        {0, 1, 2}
            Enumeration to signify underlying data type returned by the BLAS dot product calculation.
            0 signifies double, 1 signifies double, and 2 signifies that custom cython loops were used
            instead of BLAS.
    """
    pass

def score_sentence_cbow(model, sentence, _work, _neu1): # real signature unknown; restored from __doc__
    """
    score_sentence_cbow(model, sentence, _work, _neu1)
    Obtain likelihood score for a single sentence in a fitted CBOW representation.
    
        Notes
        -----
        This scoring function is only implemented for hierarchical softmax (`model.hs == 1`).
        The model should have been trained using the skip-gram model (`model.cbow` == 1`).
    
        Parameters
        ----------
        model : :class:`~gensim.models.word2vec.Word2Vec`
            The trained model. It **MUST** have been trained using hierarchical softmax and the CBOW algorithm.
        sentence : list of str
            The words comprising the sentence to be scored.
        _work : np.ndarray
            Private working memory for each worker.
        _neu1 : np.ndarray
            Private working memory for each worker.
    
        Returns
        -------
        float
            The probability assigned to this sentence by the Skip-Gram model.
    """
    pass

def score_sentence_sg(model, sentence, _work): # real signature unknown; restored from __doc__
    """
    score_sentence_sg(model, sentence, _work)
    Obtain likelihood score for a single sentence in a fitted skip-gram representation.
    
        Notes
        -----
        This scoring function is only implemented for hierarchical softmax (`model.hs == 1`).
        The model should have been trained using the skip-gram model (`model.sg` == 1`).
    
        Parameters
        ----------
        model : :class:`~gensim.models.word2vec.Word2Vec`
            The trained model. It **MUST** have been trained using hierarchical softmax and the skip-gram algorithm.
        sentence : list of str
            The words comprising the sentence to be scored.
        _work : np.ndarray
            Private working memory for each worker.
    
        Returns
        -------
        float
            The probability assigned to this sentence by the Skip-Gram model.
    """
    pass

def train_batch_cbow(model, sentences, alpha, _work, _neu1, compute_loss): # real signature unknown; restored from __doc__
    """
    train_batch_cbow(model, sentences, alpha, _work, _neu1, compute_loss)
    Update CBOW model by training on a batch of sentences.
    
        Called internally from :meth:`~gensim.models.word2vec.Word2Vec.train`.
    
        Parameters
        ----------
        model : :class:`~gensim.models.word2vec.Word2Vec`
            The Word2Vec model instance to train.
        sentences : iterable of list of str
            The corpus used to train the model.
        alpha : float
            The learning rate.
        _work : np.ndarray
            Private working memory for each worker.
        _neu1 : np.ndarray
            Private working memory for each worker.
        compute_loss : bool
            Whether or not the training loss should be computed in this batch.
    
        Returns
        -------
        int
            Number of words in the vocabulary actually used for training (They already existed in the vocabulary
            and were not discarded by negative sampling).
    """
    pass

def train_batch_sg(model, sentences, alpha, _work, compute_loss): # real signature unknown; restored from __doc__
    """
    train_batch_sg(model, sentences, alpha, _work, compute_loss)
    Update skip-gram model by training on a batch of sentences.
    
        Called internally from :meth:`~gensim.models.word2vec.Word2Vec.train`.
    
        Parameters
        ----------
        model : :class:`~gensim.models.word2Vec.Word2Vec`
            The Word2Vec model instance to train.
        sentences : iterable of list of str
            The corpus used to train the model.
        alpha : float
            The learning rate
        _work : np.ndarray
            Private working memory for each worker.
        compute_loss : bool
            Whether or not the training loss should be computed in this batch.
    
        Returns
        -------
        int
            Number of words in the vocabulary actually used for training (They already existed in the vocabulary
            and were not discarded by negative sampling).
    """
    pass

# classes

class REAL(__numpy.floating):
    """ 32-bit floating-point number. Character code 'f'. C float compatible. """
    def __abs__(self, *args, **kwargs): # real signature unknown
        """ abs(self) """
        pass

    def __add__(self, *args, **kwargs): # real signature unknown
        """ Return self+value. """
        pass

    def __and__(self, *args, **kwargs): # real signature unknown
        """ Return self&value. """
        pass

    def __bool__(self, *args, **kwargs): # real signature unknown
        """ self != 0 """
        pass

    def __divmod__(self, *args, **kwargs): # real signature unknown
        """ Return divmod(self, value). """
        pass

    def __eq__(self, *args, **kwargs): # real signature unknown
        """ Return self==value. """
        pass

    def __float__(self, *args, **kwargs): # real signature unknown
        """ float(self) """
        pass

    def __floordiv__(self, *args, **kwargs): # real signature unknown
        """ Return self//value. """
        pass

    def __ge__(self, *args, **kwargs): # real signature unknown
        """ Return self>=value. """
        pass

    def __gt__(self, *args, **kwargs): # real signature unknown
        """ Return self>value. """
        pass

    def __hash__(self, *args, **kwargs): # real signature unknown
        """ Return hash(self). """
        pass

    def __init__(self, *args, **kwargs): # real signature unknown
        pass

    def __int__(self, *args, **kwargs): # real signature unknown
        """ int(self) """
        pass

    def __invert__(self, *args, **kwargs): # real signature unknown
        """ ~self """
        pass

    def __le__(self, *args, **kwargs): # real signature unknown
        """ Return self<=value. """
        pass

    def __lshift__(self, *args, **kwargs): # real signature unknown
        """ Return self<<value. """
        pass

    def __lt__(self, *args, **kwargs): # real signature unknown
        """ Return self<value. """
        pass

    def __mod__(self, *args, **kwargs): # real signature unknown
        """ Return self%value. """
        pass

    def __mul__(self, *args, **kwargs): # real signature unknown
        """ Return self*value. """
        pass

    def __neg__(self, *args, **kwargs): # real signature unknown
        """ -self """
        pass

    @staticmethod # known case of __new__
    def __new__(*args, **kwargs): # real signature unknown
        """ Create and return a new object.  See help(type) for accurate signature. """
        pass

    def __ne__(self, *args, **kwargs): # real signature unknown
        """ Return self!=value. """
        pass

    def __or__(self, *args, **kwargs): # real signature unknown
        """ Return self|value. """
        pass

    def __pos__(self, *args, **kwargs): # real signature unknown
        """ +self """
        pass

    def __pow__(self, *args, **kwargs): # real signature unknown
        """ Return pow(self, value, mod). """
        pass

    def __radd__(self, *args, **kwargs): # real signature unknown
        """ Return value+self. """
        pass

    def __rand__(self, *args, **kwargs): # real signature unknown
        """ Return value&self. """
        pass

    def __rdivmod__(self, *args, **kwargs): # real signature unknown
        """ Return divmod(value, self). """
        pass

    def __repr__(self, *args, **kwargs): # real signature unknown
        """ Return repr(self). """
        pass

    def __rfloordiv__(self, *args, **kwargs): # real signature unknown
        """ Return value//self. """
        pass

    def __rlshift__(self, *args, **kwargs): # real signature unknown
        """ Return value<<self. """
        pass

    def __rmod__(self, *args, **kwargs): # real signature unknown
        """ Return value%self. """
        pass

    def __rmul__(self, *args, **kwargs): # real signature unknown
        """ Return value*self. """
        pass

    def __ror__(self, *args, **kwargs): # real signature unknown
        """ Return value|self. """
        pass

    def __rpow__(self, *args, **kwargs): # real signature unknown
        """ Return pow(value, self, mod). """
        pass

    def __rrshift__(self, *args, **kwargs): # real signature unknown
        """ Return value>>self. """
        pass

    def __rshift__(self, *args, **kwargs): # real signature unknown
        """ Return self>>value. """
        pass

    def __rsub__(self, *args, **kwargs): # real signature unknown
        """ Return value-self. """
        pass

    def __rtruediv__(self, *args, **kwargs): # real signature unknown
        """ Return value/self. """
        pass

    def __rxor__(self, *args, **kwargs): # real signature unknown
        """ Return value^self. """
        pass

    def __str__(self, *args, **kwargs): # real signature unknown
        """ Return str(self). """
        pass

    def __sub__(self, *args, **kwargs): # real signature unknown
        """ Return self-value. """
        pass

    def __truediv__(self, *args, **kwargs): # real signature unknown
        """ Return self/value. """
        pass

    def __xor__(self, *args, **kwargs): # real signature unknown
        """ Return self^value. """
        pass


# variables with complex values

__loader__ = None # (!) real value is '<_frozen_importlib_external.ExtensionFileLoader object at 0x7f07e21ba6d8>'

__pyx_capi__ = {
    'EXP_TABLE': None, # (!) real value is '<capsule object "__pyx_t_6gensim_6models_14word2vec_inner_REAL_t [0x3E8]" at 0x7f07e21c63c0>'
    'bisect_left': None, # (!) real value is '<capsule object "unsigned PY_LONG_LONG (__pyx_t_5numpy_uint32_t *, unsigned PY_LONG_LONG, unsigned PY_LONG_LONG, unsigned PY_LONG_LONG)" at 0x7f07e21c6510>'
    'dsdot': None, # (!) real value is '<capsule object "__pyx_t_6gensim_6models_14word2vec_inner_dsdot_ptr" at 0x7f07e21c6330>'
    'init_w2v_config': None, # (!) real value is '<capsule object "PyObject *(struct __pyx_t_6gensim_6models_14word2vec_inner_Word2VecConfig *, PyObject *, PyObject *, PyObject *, PyObject *, struct __pyx_opt_args_6gensim_6models_14word2vec_inner_init_w2v_config *__pyx_optional_args)" at 0x7f07e21c6630>'
    'our_dot': None, # (!) real value is '<capsule object "__pyx_t_6gensim_6models_14word2vec_inner_our_dot_ptr" at 0x7f07e21c63f0>'
    'our_dot_double': None, # (!) real value is '<capsule object "__pyx_t_6gensim_6models_14word2vec_inner_REAL_t (int const *, float const *, int const *, float const *, int const *)" at 0x7f07e21c6450>'
    'our_dot_float': None, # (!) real value is '<capsule object "__pyx_t_6gensim_6models_14word2vec_inner_REAL_t (int const *, float const *, int const *, float const *, int const *)" at 0x7f07e21c6480>'
    'our_dot_noblas': None, # (!) real value is '<capsule object "__pyx_t_6gensim_6models_14word2vec_inner_REAL_t (int const *, float const *, int const *, float const *, int const *)" at 0x7f07e21c64b0>'
    'our_saxpy': None, # (!) real value is '<capsule object "__pyx_t_6gensim_6models_14word2vec_inner_our_saxpy_ptr" at 0x7f07e21c6420>'
    'our_saxpy_noblas': None, # (!) real value is '<capsule object "void (int const *, float const *, float const *, int const *, float *, int const *)" at 0x7f07e21c64e0>'
    'random_int32': None, # (!) real value is '<capsule object "unsigned PY_LONG_LONG (unsigned PY_LONG_LONG *)" at 0x7f07e21c6540>'
    'saxpy': None, # (!) real value is '<capsule object "__pyx_t_6gensim_6models_14word2vec_inner_saxpy_ptr" at 0x7f07e21c62d0>'
    'scopy': None, # (!) real value is '<capsule object "__pyx_t_6gensim_6models_14word2vec_inner_scopy_ptr" at 0x7f07e21c62a0>'
    'sdot': None, # (!) real value is '<capsule object "__pyx_t_6gensim_6models_14word2vec_inner_sdot_ptr" at 0x7f07e21c6300>'
    'snrm2': None, # (!) real value is '<capsule object "__pyx_t_6gensim_6models_14word2vec_inner_snrm2_ptr" at 0x7f07e21c6360>'
    'sscal': None, # (!) real value is '<capsule object "__pyx_t_6gensim_6models_14word2vec_inner_sscal_ptr" at 0x7f07e21c6390>'
    'w2v_fast_sentence_cbow_hs': None, # (!) real value is '<capsule object "void (__pyx_t_5numpy_uint32_t const *, __pyx_t_5numpy_uint8_t const *, int *, __pyx_t_6gensim_6models_14word2vec_inner_REAL_t *, __pyx_t_6gensim_6models_14word2vec_inner_REAL_t *, __pyx_t_6gensim_6models_14word2vec_inner_REAL_t *, int const , __pyx_t_5numpy_uint32_t const *, __pyx_t_6gensim_6models_14word2vec_inner_REAL_t const , __pyx_t_6gensim_6models_14word2vec_inner_REAL_t *, int, int, int, int, __pyx_t_6gensim_6models_14word2vec_inner_REAL_t *, int const , __pyx_t_6gensim_6models_14word2vec_inner_REAL_t *)" at 0x7f07e21c65d0>'
    'w2v_fast_sentence_cbow_neg': None, # (!) real value is '<capsule object "unsigned PY_LONG_LONG (int const , __pyx_t_5numpy_uint32_t *, unsigned PY_LONG_LONG, int *, __pyx_t_6gensim_6models_14word2vec_inner_REAL_t *, __pyx_t_6gensim_6models_14word2vec_inner_REAL_t *, __pyx_t_6gensim_6models_14word2vec_inner_REAL_t *, int const , __pyx_t_5numpy_uint32_t const *, __pyx_t_6gensim_6models_14word2vec_inner_REAL_t const , __pyx_t_6gensim_6models_14word2vec_inner_REAL_t *, int, int, int, int, unsigned PY_LONG_LONG, __pyx_t_6gensim_6models_14word2vec_inner_REAL_t *, int const , __pyx_t_6gensim_6models_14word2vec_inner_REAL_t *)" at 0x7f07e21c6600>'
    'w2v_fast_sentence_sg_hs': None, # (!) real value is '<capsule object "void (__pyx_t_5numpy_uint32_t const *, __pyx_t_5numpy_uint8_t const *, int const , __pyx_t_6gensim_6models_14word2vec_inner_REAL_t *, __pyx_t_6gensim_6models_14word2vec_inner_REAL_t *, int const , __pyx_t_5numpy_uint32_t const , __pyx_t_6gensim_6models_14word2vec_inner_REAL_t const , __pyx_t_6gensim_6models_14word2vec_inner_REAL_t *, __pyx_t_6gensim_6models_14word2vec_inner_REAL_t *, int const , __pyx_t_6gensim_6models_14word2vec_inner_REAL_t *)" at 0x7f07e21c6570>'
    'w2v_fast_sentence_sg_neg': None, # (!) real value is '<capsule object "unsigned PY_LONG_LONG (int const , __pyx_t_5numpy_uint32_t *, unsigned PY_LONG_LONG, __pyx_t_6gensim_6models_14word2vec_inner_REAL_t *, __pyx_t_6gensim_6models_14word2vec_inner_REAL_t *, int const , __pyx_t_5numpy_uint32_t const , __pyx_t_5numpy_uint32_t const , __pyx_t_6gensim_6models_14word2vec_inner_REAL_t const , __pyx_t_6gensim_6models_14word2vec_inner_REAL_t *, unsigned PY_LONG_LONG, __pyx_t_6gensim_6models_14word2vec_inner_REAL_t *, int const , __pyx_t_6gensim_6models_14word2vec_inner_REAL_t *)" at 0x7f07e21c65a0>'
}

__spec__ = None # (!) real value is "ModuleSpec(name='gensim.models.word2vec_inner', loader=<_frozen_importlib_external.ExtensionFileLoader object at 0x7f07e21ba6d8>, origin='/data/apps/anaconda3/envs/d3m/lib/python3.6/site-packages/gensim/models/word2vec_inner.cpython-36m-x86_64-linux-gnu.so')"

__test__ = {}

