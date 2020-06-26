# Installation and documentation at https://gitlab.com/dmartinez05/d3m_primitive_profiler
from d3m import index

from D3MPrimitiveProfiler.TestClassificationPrimitive import TestPrimitive, pretty_print

primitives = [
    'd3m.primitives.classification.cover_tree.Fastlvm',
    'd3m.primitives.clustering.gmm.Fastlvm',
    'd3m.primitives.clustering.k_means.Fastlvm',
    'd3m.primitives.natural_language_processing.glda.Fastlvm',
    'd3m.primitives.natural_language_processing.hdp.Fastlvm',
    'd3m.primitives.natural_language_processing.lda.Fastlvm',
    'd3m.primitives.regression.cover_tree.Fastlvm',
]


def test(primitive_name: str):
    primitive = index.get_primitive(primitive_name)

    test = TestPrimitive(primitive=primitive)
    test.basic_test()
    test.scalability_test()
    test.hyperparameter_importance()
    pretty_print(test.attributes['summary'])


for primitive in primitives:
    test(primitive)
