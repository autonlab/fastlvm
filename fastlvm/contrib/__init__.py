"""This package contains interfaces and functionality to compute pair-wise document similarities within a corpus
of documents.
"""

import logging

__version__ = '3.8.1'


logger = logging.getLogger('gensim')
if len(logger.handlers) == 0:  # To ensure reload() doesn't add another one
    logger.addHandler(logging.NullHandler())
