__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.optimization.optimizer import ContinuousOptimizer


class DistributionBasedOptimizer(ContinuousOptimizer):
    """ The parent class for all optimization algorithms that are based on
    iteratively updating a search distribution.

    Provides a number of potentially useful methods that could be used by subclasses. """

    online = False
    batchSize = 100

    # distribution types
    GAUSSIAN = 1
    CAUCHY = 2
    GENERALIZEDGAUSSIAN = 3
    STUDENTT = 4

    distributionType = GAUSSIAN

    storeAllDistributions = False

    def _updateDistribution(self, dparamDeltas):
        """ Update the parameters of the current distribution, directly. """

    def _generateSample(self):
        """ Generate 1 sample from the current distribution. """

    def _generateConformingBatch(self):
        """ Generate a batch of samples that conforms to the current distribution.
        If importance mixing is enabled, this can reuse old samples. """



