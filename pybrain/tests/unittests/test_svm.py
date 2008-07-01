"""
    >>> from numpy import array

    >>> from pybrain.datasets import SupervisedDataSet

    >>> from pybrain.supervised.trainers.svm import SVMTrainer

    >>> from pybrain.structure.modules.svm import KT,SVM

    >>> dataset = SupervisedDataSet(2,1)

    >>> dataset.addSample( [ 0.10, 0.50 ], [2] )

    >>> dataset.addSample( [ 1.20, 1.60 ], [2] )

    >>> dataset.addSample( [ 0.20, 0.80 ], [2] )

    >>> dataset.addSample( [ 0.30, 1.70 ], [8] )

    >>> dataset.addSample( [ 1.40, 0.10 ], [8] )

    >>> module  = SVM( dataset.indim, dataset.outdim, KT.RBF )

    >>> trainer = SVMTrainer(module, cost=20.)

    >>> trainer.setData( dataset )

    >>> trainer.train()

    >>> module.classify([ 0.10, 0.50 ])
    2.0

    >>> module.classify([ 1.20, 1.60 ])
    2.0

    >>> module.classify([ 0.20, 0.80 ])
    2.0

    >>> module.classify([ 0.30, 1.70 ])
    8.0

    >>> module.classify([ 1.40, 0.10 ])
    8.0

"""


__author__ = 'Michael Isik, isikmichael@gmx.net'

from pybrain.tests import runModuleTestSuite

if __name__ == "__main__":
    runModuleTestSuite(__import__('__main__'))

