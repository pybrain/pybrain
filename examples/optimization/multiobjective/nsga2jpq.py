from __future__ import print_function

#!/usr/bin/env python
""" An illustration of using the NSGA-II multi-objective optimization algorithm
on Unconstrained Multi-Objective Optimization benchmark function. """

__author__ = 'Jean Pierre Queau, jeanpierre.queau@sbmoffshore.com'

from pybrain.optimization import MultiObjectiveGA
from pybrain.rl.environments.functions.multiobjective import Deb, Pol
import pylab
from scipy import zeros, array

# The Deb function
#f = Deb()
# The Pol function
f = Pol()

# start at the origin
x0 = zeros(f.indim)

x0 = array([min_ for min_, max_ in f.xbound])

# the optimization for a maximum of 25 generations
n = MultiObjectiveGA(f, x0, storeAllEvaluations = True, populationSize = 50, eliteProportion = 1.0,
    topProportion = 1.0, mutationProb = 0.5, mutationStdDev = 0.1, storeAllPopulations = True, allowEquality = False)
print('Start Learning')
n.learn(30)
print('End Learning')

# plotting the results (blue = all evaluated points, red = resulting pareto front)
print('Plotting the Results')
print('All Evaluations')
for x in n._allEvaluations: pylab.plot([x[0]], [x[1]], 'b.')
for x in n.bestEvaluation: pylab.plot([x[0]], [x[1]], 'ro')
pylab.show()
print('Pareto Front')
for x in n.bestEvaluation: pylab.plot([x[0]], [x[1]], 'ro')
pylab.show()
print('===========')
print('= Results =') 
print('===========')
'''
i=0
for gen in n._allGenerations:
    print 'Generation: ',i
    for j in range(len(gen[1])):
        print gen[1].keys()[j],gen[1].values()[j]
    i+=1
'''
print('Population size ',n.populationSize)
print('Elitism Proportion ',n.eliteProportion)
print('Mutation Probability ',n.mutationProb)
print('Mutation Std Deviation ',n.mutationStdDev)
print('Objective Evaluation number ',n.numEvaluations)
print('last generation Length of bestEvaluation ',len(n.bestEvaluation))
print('Best Evaluable : Best Evaluation')
for i in range(len(n.bestEvaluation)):
    assert len(n.bestEvaluation) == len(n.bestEvaluable)
    print(n.bestEvaluable[i],':',n.bestEvaluation[i])