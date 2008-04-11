__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.rl.evolvables import MaskedParameters
from pybrain.tools.shortcuts import buildNetwork
from pybrain.rl.learners.searchprocesses import MemeticHillClimber

from pybrain.examples.datasets import XORDataSet


ds = XORDataSet()
evaluator = lambda m: -ds.evaluateModuleMSE(m)

initm = MaskedParameters(buildNetwork(ds.indim, 2, ds.outdim), maxComplexity = 7)

sp = MemeticHillClimber(initm, evaluator)

sp.search(1000, verbose = True)
