## v3.0.0
* Two new primitives
    * `d3m.primitives.regression.cover_tree.Fastlvm`
    * `d3m.primitives.classification.cover_tree.Fastlvm`
* Removed a primitive
    * `d3m.primitives.clustering.cover_tree.Fastlvm`

## v2.0.0
### Enhancements
* Changed the primitive namespace per [Issue #3](https://gitlab.com/datadrivendiscovery/d3m/issues/3)
    * `d3m.primitives.cmu.fastlvm.CoverTree` -> `d3m.primitives* .clustering.cover_tree.Fastlvm`
    * `d3m.primitives.cmu.fastlvm.KMeans` -> `d3m.primitives.clustering* .k_means.Fastlvm`
    * `d3m.primitives.cmu.fastlvm.GMM` -> `d3m.primitives.clustering.gmm* .Fastlvm`
    * `d3m.primitives.cmu.fastlvm.LDA` -> `d3m.primitives* .natural_language_processing.lda.Fastlvm`
    * `d3m.primitives.cmu.fastlvm.GLDA` -> `d3m.primitives.natural_language_processing.glda.Fastlvm`
    * `d3m.primitives.cmu.fastlvm.HDP` -> `d3m.primitives.natural_language_processing.hdp.Fastlvm`
