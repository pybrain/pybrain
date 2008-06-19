""" Simple script to do individual tests of FEM and CMA on linear functions. """

__author__ = 'Tom Schaul'

from scipy import randn, log10, array

from pybrain.utilities import storeCallResults
from pybrain.rl.learners import CMAES, FEM
from pybrain.rl.environments.functions import RotateFunction, TranslateFunction, LinearFunction, OppositeFunction
from pybrain.tools.rankingfunctions import TopLinearRanking, ExponentialRanking, SmoothGiniRanking, TopSelection, RankingFunction

dim = 3
basef = LinearFunction(dim)
f = TranslateFunction(RotateFunction(basef))
f.desiredValue = 1e6
x0 = randn(dim)
#cma = CMAES(f, x0)
fem = FEM(f, x0,
         batchsize = 50, 
         onlineLearning = False,
         forgetFactor = 0.03,
         elitist = False,
         #rankingFunction = RankingFunction(),               # fail
         #rankingFunction = TopSelection(topFraction = 0.02),# 15
         #rankingFunction = TopSelection(topFraction = 0.1), # 25
         #rankingFunction = TopSelection(topFraction = 0.3), # 55
         #rankingFunction = TopSelection(topFraction = 0.5), # fail
         #rankingFunction = TopLinearRanking(topFraction = 0.1), # 20
         #rankingFunction = TopLinearRanking(topFraction = 0.5), # 55
         #rankingFunction = TopLinearRanking(topFraction = 0.7), # 100
         #rankingFunction = TopLinearRanking(topFraction = 0.9), # fail
         #rankingFunction = ExponentialRanking(temperature = 1.),# fail
         #rankingFunction = ExponentialRanking(temperature =10.),# 25
         #rankingFunction = SmoothGiniRanking(gini = 0.01, linearComponent = 0.), # 15
         #rankingFunction = SmoothGiniRanking(gini = 0.1, linearComponent = 0.),  # 20
         #rankingFunction = SmoothGiniRanking(gini = 0.7, linearComponent = 0.),  # 150
         #rankingFunction = SmoothGiniRanking(gini = 0.1, linearComponent = 0.1), # fail
         maxEvaluations = 10000,
         )
                        
res = storeCallResults(f)


print 'FEM:', fem.learn(), len(res)

#print 'CMA:', cma.learn(), len(res)

if True:
    import pylab
    #pylab.plot(log10(-array(res)))
    pylab.plot(log10(-array(fem.muevals)))
    #pylab.figure()
    #pylab.plot(log10(-array(res)))
    pylab.show()
