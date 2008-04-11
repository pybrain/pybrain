

from scipy import rand

from pybrain.examples.datasets import XORDataSet
from pybrain.rl.evolvables.evolvable import Evolvable
from pybrain.tools.shortcuts import buildNetwork
from pybrain.rl.learners.searchprocesses.randomsearch import RandomSearch
from pybrain.rl.learners.searchprocesses.hillclimber import HillClimber

def r(self):
    self.module.params *= 0
    self.module.params += -2+4*rand(self.module.paramdim)

Evolvable.randomize = r

ds = XORDataSet()
#evaluator = lambda m: -ds.evaluateModuleMSE(m)
def ee(m):
    print 'Eval:', m.module.params,
    res = -ds.evaluateModuleMSE(m.module)
    print res
    return res

initm = Evolvable(buildNetwork(ds.indim, 1, ds.outdim, bias = False))

sp = RandomSearch(initm, ee)
#sp = HillClimber(initm, evaluator)
sp.search(1000, verbose = True)
