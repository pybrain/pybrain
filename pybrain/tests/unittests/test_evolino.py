"""
    >>> import numpy
    >>> from pybrain.datasets.sequential              import SequentialDataSet
    >>> from pybrain.structure.modules.evolinonetwork import EvolinoNetwork
    >>> from pybrain.supervised.trainers.evolino      import EvolinoTrainer

    >>> dataset = SequentialDataSet(0,1)
    >>> dataset.newSequence()
    >>> for x in numpy.arange( 0 , 30 , 0.2 ):
    ...     dataset.addSample([], [x])
    ...


Tests for Construction of Module and Trainer
--------------------------------------------
    >>> net = EvolinoNetwork( dataset.outdim, 7 )
    >>> trainer = EvolinoTrainer(
    ...     net,
    ...     dataset=dataset,
    ...     subPopulationSize = 5,
    ...     nParents = 2,
    ...     nCombinations = 1,
    ...     initialWeightRange = ( -0.01 , 0.01 ),
    ...     backprojectionFactor = 0.001,
    ...     mutationAlpha = 0.001,
    ...     nBurstMutationEpochs = numpy.Infinity,
    ...     wtRatio = 1./3.,
    ...     verbosity = 0)


Tests for Training and Applying
-------------------------------
    >>> trainer.trainEpochs( 1 )
    >>> trainer.evaluation.max_fitness > -100
    True

    >>> sequence = dataset.getField('target')
    >>> net.extrapolate(sequence, len(sequence))[-1][0]>55
    True
"""



__author__ = 'Michael Isik, isikmichael@gmx.net'

from pybrain.tests import runModuleTestSuite

if __name__ == "__main__":
    runModuleTestSuite(__import__('__main__'))

