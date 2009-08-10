""" An illustration of using the NSGA-II multi-objective optimization algorithm 
on a simple standard benchmark function. """

__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.optimization import MultiObjectiveGA
from pybrain.rl.environments.functions.multiobjective import KurBenchmark
from pybrain.utilities import storeCallResults
import pylab
from scipy import zeros
       
# The benchmark function
f = KurBenchmark()

# keep track of all evaluations
res = storeCallResults(f)

# start at the origin
x0 = zeros(f.indim)

# the optimization for a maximum of 2500 function evaluations
n = MultiObjectiveGA(f, x0)
n.learn(2500)

# plotting the results (blue = all evaluated points, red = resulting pareto front)
for x in res: pylab.plot([x[1]], [x[0]], 'b+')
for x in n.bestEvaluation: pylab.plot([x[1]], [x[0]], 'ro')
pylab.show()