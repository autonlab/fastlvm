import copy
import numpy as np
import utilsc
import typing
from d3m.metadata import base as metadata_base

_stirling = utilsc.new_stirling()

def read_corpus(fname, vocab=[], stopwords=[]):
    return utilsc.read_corpus(fname, vocab, stopwords)
    
def get_ref_count(var):
    return utilsc.ref_count(var)

def kmeanspp(k, points):
    seed_idx = utilsc.kmeanspp(k, points)
    seeds = points[seed_idx]
    return seeds

def log_stirling_num(n, m):
    return utilsc.log_stirling_num(_stirling, n, m)

def uratio(n, m):
    return utilsc.uratio(_stirling, n, m)

def vratio(n, m):
    return utilsc.vratio(_stirling, n, m)

def wratio(n, m):
    return utilsc.wratio(_stirling, n, m)


# Copied from common_primitives.utils
def list_columns_with_semantic_types(metadata: metadata_base.DataMetadata,
                                      semantic_types: typing.Sequence[str], *,
                                      at: metadata_base.Selector = ()) -> typing.Sequence[int]:
    """
    This is similar to ``get_columns_with_semantic_type``, but it returns all column indices
    for a dimension instead of ``ALL_ELEMENTS`` element.

    Moreover, it operates on a list of semantic types, where a column is returned
    if it matches any semantic type on the list.
    """

    columns = []

    for element in metadata.get_elements(list(at) + [metadata_base.ALL_ELEMENTS]):
        metadata_semantic_types = metadata.query(list(at) + [metadata_base.ALL_ELEMENTS, element]).get('semantic_types', ())
        # TODO: Should we handle inheritance between semantic types here?
        if any(semantic_type in metadata_semantic_types for semantic_type in semantic_types):
            if element is metadata_base.ALL_ELEMENTS:
                return list(range(metadata.query(list(at) + [metadata_base.ALL_ELEMENTS]).get('dimension', {}).get('length', 0)))
            else:
                columns.append(typing.cast(int, element))

    return columns


def split_inputs(tokenized, frac):
    """Uniformly split the data to training and validation
    :returns a tuple of training and validation
    """
    num_training = int(round((1 - frac) * len(tokenized)))
    num_training = 1 if num_training == 0 else num_training
    permutation = np.random.permutation(np.arange(len(tokenized)))
    training = tokenized[permutation[:num_training]]
    if num_training == len(tokenized):  # self._frac == 0
        validation = training
    else:
        validation = tokenized[permutation[(num_training+1):]]
    return training, validation


def get_documents(training_inputs):
    """Extract the text columns and concatenate them row-wise

    returns: a Series. Each element is a string.
    """
    # Adapted from https://github.com/brekelma/dsbox_corex/blob/master/corex_text.py

    # Get the text columns
    text_attributes = list_columns_with_semantic_types(
        metadata=training_inputs.metadata,
        semantic_types=["http://schema.org/Text"])
    all_attributes = list_columns_with_semantic_types(
        metadata=training_inputs.metadata,
        semantic_types=["https://metadata.datadrivendiscovery.org/types/Attribute"])
    categorical_attributes = list_columns_with_semantic_types(
        metadata=training_inputs.metadata,
        semantic_types=["https://metadata.datadrivendiscovery.org/types/CategoricalData"])

    # want text columns that are attributes
    text_columns = set(all_attributes).intersection(text_attributes)

    # but, don't want to edit categorical columns
    text_columns = set(text_columns) - set(categorical_attributes)

    # and, we want the text columns as a list
    text_columns = list(text_columns)

    # TODO if no text columns are present don't do anything
    if len(text_columns) == 0:
        raise ValueError("The input has no text columns.")

    # concatenate the columns row-wise
    raw_documents = None
    for column_index in text_columns:
        if raw_documents is not None:
            raw_documents = raw_documents.str.cat(training_inputs.iloc[:, column_index], sep=" ")
        else:
            raw_documents = copy.deepcopy(training_inputs.iloc[:, column_index])

    return raw_documents


def tokenize(raw_documents, vocabulary, analyze):
    """Tokenize the raw documents

    Returns a ndarray. Each element is an ndarray of items of unit32 type. The ndarray can be of different length.
    """
    if vocabulary is None or analyze is None:
        return []

    tokenized = []
    for doc in raw_documents:
        row = []
        for feature in analyze(doc):
            try:
                feature_idx = vocabulary[feature]
                row.append(feature_idx)
            except KeyError:
                # Ignore out-of-vocabulary items for fixed_vocab=True
                continue
        tokenized.append(np.array(row, dtype=np.uint32))

    return np.array(tokenized)


def tpd(zs, k):
    """ Convert to feature vector
    Returns a 2D ndarray
    """
    tpdm = np.zeros((len(zs), k))
    for i, doc in enumerate(zs):
        for z in doc:
            if z < k:
                tpdm[i, z] += 1
        if len(doc) > 0:
            tpdm[i] /= len(doc)
    return tpdm

