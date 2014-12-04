from __future__ import print_function

#!/usr/bin/env python
""" An illustration of using the NSGA-II multi-objective optimization algorithm
on Constrained Multi-Objective Optimization benchmark function. """

__author__ = 'Jean Pierre Queau, jeanpierre.queau@sbmoffshore.com'

from pybrain.optimization import ConstMultiObjectiveGA
from pybrain.rl.environments.functions.multiobjective import ConstDeb,ConstSrn, \
     ConstOsy,ConstTnk,ConstBnh
import pylab
from scipy import zeros, array

# The Deb function
#f = ConstDeb()
# The Srinivas & Deb function
#f = ConstSrn()
# The Osyczka & Kundu function
#f = ConstOsy()
# The Tanaka function
#f = ConstTnk()
# The Binh & Korn function
f = ConstBnh()
# start at the origin
x0 = zeros(f.indim)

x0 = array([min_ for min_, max_ in f.xbound])

# the optimization for a maximum of 25 generations
n = ConstMultiObjectiveGA(f, x0, storeAllEvaluations = True, populationSize = 100, eliteProportion = 1.0,
    topProportion = 1.0, mutationProb = 1.0, mutationStdDev = 0.3, storeAllPopulations = True, allowEquality = False)
print('Start Learning')
n.learn(50)
print('End Learning')
# plotting the results (blue = all evaluated points, red = resulting pareto front)
print('Plotting the Results')
print('All Evaluations.... take some time')
for x in n._allEvaluations:
    if x[1]:
        pylab.plot([x[0][0]], [x[0][1]], 'b.')
    else:
        pylab.plot([x[0][0]], [x[0][1]], 'r.')
for x in n.bestEvaluation: pylab.plot([x[0][0]], [x[0][1]], 'go')
pylab.show()
print('Pareto Front')
for x in n.bestEvaluation: pylab.plot([x[0][0]], [x[0][1]], 'go')
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