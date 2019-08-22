import numpy as np
from d3m import container
from d3m.metadata import base as metadata_base
from unittest import TestCase
from fastlvm import read_corpus
from fastlvm.lda import HyperParams
from fastlvm import LDA


class TestLDA(TestCase):
    def setUp(self) -> None:
        pass

    def test_produce(self):
        # Load NIPS data
        trngdata, vocab = read_corpus('../data/nips/corpus.train')
        testdata, vocab = read_corpus('../data/nips/corpus.test', vocab)

        # Init LDA model
        num_topics = 10
        hp = HyperParams(k=num_topics, iters=100, num_top=1, seed=1, frac=0.01)
        canlda = LDA(hyperparams=hp)
        canlda.set_training_data(inputs=self.transform(trngdata))

        canlda.fit()
        # Test on held out data using learned model
        a = canlda.evaluate(inputs=testdata)

        # Get topic matrix
        tm = canlda.produce_topic_matrix()

        # Get topic assignment
        zz = canlda.produce(inputs=self.transform(testdata))
        zz = zz.value
        self.assertEqual(len(zz), len(testdata))

        id2word = {idx: v for idx, v in enumerate(vocab)}
        # Read word|topic distribution from gensim
        m = self.baseline(np.array(trngdata), id2word=id2word, num_topics=num_topics)

        # use the baseline weight
        np.copyto(tm, m[tm.shape[0], :])  # FIXME: baseline and LDA weight matrices have different shape

        # Test on held out data using gensim model
        b = canlda.evaluate(inputs=testdata)

        self.assertAlmostEqual(a, b, places=1)

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
