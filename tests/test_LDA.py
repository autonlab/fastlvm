import pickle

import numpy as np
from d3m import container
from d3m.metadata import base as metadata_base
from unittest import TestCase
from fastlvm import read_corpus
from fastlvm.lda import HyperParams
from fastlvm import LDA


class TestLDA(TestCase):
    def setUp(self) -> None:
        self.num_topics = 10
        # Load NIPS data
        self.trngdata, self.vocab = read_corpus('../data/nips/corpus.train')
        self.testdata, self.vocab = read_corpus('../data/nips/corpus.test', self.vocab)

        self.canlda = None  # LDA model

    def lda(self, trngdata, testdata):
        # Init LDA model
        hp = HyperParams(k=self.num_topics, iters=100, num_top=1, frac=0.01)
        canlda = LDA(hyperparams=hp, random_seed=0)
        canlda.set_training_data(inputs=self.transform(trngdata))

        canlda.fit()
        # Test on held out data using learned model
        a = canlda.evaluate(inputs=testdata)

        self.canlda = canlda
        return a

    def test_produce(self):
        a = self.lda(trngdata=self.trngdata, testdata=self.testdata)
        self.assertTrue(a is not None)

    def test_compare_to_baseline(self):
        a = self.lda(trngdata=self.trngdata, testdata=self.testdata)

        # Get topic matrix
        tm = self.canlda.produce_topic_matrix()

        # Get topic assignment
        zz = self.canlda.produce(inputs=self.transform(self.testdata))
        zz = zz.value
        self.assertEqual(len(zz), len(self.testdata))

        id2word = {idx: v for idx, v in enumerate(self.vocab)}
        # Read word|topic distribution from gensim
        m = self.baseline(np.array(self.trngdata), id2word=id2word, num_topics=self.num_topics)

        # use the baseline weight
        np.copyto(tm, m[tm.shape[0], :])  # FIXME: baseline and LDA weight matrices have different shape

        # Test on held out data using gensim model
        b = self.canlda.evaluate(inputs=self.testdata)

        self.assertAlmostEqual(a, b, places=1)

    def test_pickle(self):
        """
        This calls get_params and set_params
        :return:
        """
        self.lda(trngdata=self.trngdata, testdata=self.testdata)
        b = self.canlda.evaluate(inputs=self.testdata)

        f_string = pickle.dumps(self.canlda)
        ct_new = pickle.loads(f_string)

        b_new = ct_new.evaluate(inputs=self.testdata)

        self.assertAlmostEqual(b, b_new, delta=0.1)

    def transform(self, corpus):
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

    @staticmethod
    def baseline(corpus, id2word=None, num_topics=10, filename='../data/nips/lda_gensim.npy'):
        """
        Train gensim model and returns word|topic distribution
        :param filename: name of the file that stores word|topic matrix
        :param id2word:
        :param num_topics:
        :param corpus:
        :return:
        """
        try:
            return np.load(filename)
        except IOError:
            from gensim.models.ldamodel import LdaModel
            from gensim.matutils import Dense2Corpus
            data = Dense2Corpus(corpus)
            lda = LdaModel(data, num_topics=num_topics, id2word=id2word)
            topics = lda.get_topics()
            topics = topics.T
            np.save(filename, topics)
            return topics
