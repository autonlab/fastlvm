import numpy as np
from d3m import container
from d3m.metadata import base as metadata_base
from unittest import TestCase
from fastlvm import read_corpus, LDA, HDP
from fastlvm.hdp import HyperParams


class TestHDP(TestCase):
    def test_produce(self):
        # Load NIPS data
        trngdata, vocab = read_corpus('../data/nips/corpus.train')
        testdata, vocab = read_corpus('../data/nips/corpus.test', vocab)

        # Init HDP model
        num_topics = 10
        hp = HyperParams(k=num_topics, iters=100, num_top=1, seed=1, frac=0.01)
        hdp = HDP(hyperparams=hp)
        hdp.set_training_data(inputs=self.transform(trngdata))

        hdp.fit()
        # Test on held out data using learned model
        a = hdp.evaluate(inputs=testdata)

        # TODO is it a good idea to use LDA as the baseline?
        # Use LDA model as baseline
        hp = HyperParams(k=num_topics, iters=100, num_top=1, seed=1, frac=0.01)
        canlda = LDA(hyperparams=hp)
        canlda.set_training_data(inputs=self.transform(trngdata))
        canlda.fit()
        # Test on held out data using learned model
        b = canlda.evaluate(inputs=testdata)

        self.assertAlmostEqual(a, b, places=1)

    @staticmethod
    def transform(corpus):
        """
        Convert corpus to D3M dataframe of shape Nx1
        N is the number of sentences.
        The column is of vary length with metadata text.

        :param corpus: list of ndarray, each element of the ndarray is a number representing a word
        :return:
        """
        text = []
        for sentence in corpus:
            text.append(" ".join(sentence.astype(str)))
        df = container.DataFrame(text, generate_metadata=True)

        # create metadata for the text feature columns
        for column_index in range(df.shape[1]):
            col_dict = dict(df.metadata.query((metadata_base.ALL_ELEMENTS, column_index)))
            col_dict['structural_type'] = type(1.0)
            col_dict['name'] = 'fastlvm_' + str(column_index)
            col_dict['semantic_types'] = ('http://schema.org/Text',
                                          'https://metadata.datadrivendiscovery.org/types/Attribute')
            df.metadata = df.metadata.update((metadata_base.ALL_ELEMENTS, column_index), col_dict)
        return df
